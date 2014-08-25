 uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;
uniform sampler2D texture;
uniform sampler2D t_audio;

uniform vec4 spheres[25];
uniform vec3 sphereCols[25];

varying vec2 vUv;




vec2 hash2( float n ){
  return fract(sin(vec2(n,n+1.0))*vec2(43758.5453123,22578.1459123));
}   



//------------------------------------------------------------------------
// Camera
//
// Move the camera. In this case it's using time and the mouse position
// to orbitate the camera around the origin of the world (0,0,0), where
// the yellow sphere is.
//------------------------------------------------------------------------
void doCamera( out vec3 camPos, out vec3 camTar, in float time, in float mouseX ){
  float an = 0.3 + 10.0*mouseX;
  camPos = vec3(3.5*sin(an),1.0,3.5*cos(an));
  camTar = vec3(0.0,0.0,0.0);
}


//------------------------------------------------------------------------
// Background 
//
// The background color. In this case it's just a black color.
//------------------------------------------------------------------------
vec3 doBackground( void ){
  return vec3( 0.0, 0.0, 0.0);
}

//------------------------------------------------------------------------
// Modelling 
//
// Defines the shapes (a sphere in this case) through a distance field, in
// this case it's a sphere of radius 1.
//------------------------------------------------------------------------
float doModel( vec3 p , vec3 pos ){  
  return length(p-pos) - .5;
}



float iSphere( in vec3 ro, in vec3 rd, in vec4 sph ){
 
  vec3 oc = ro - sph.xyz;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - sph.w*sph.w;
  float h = b*b - c;
  if( h<0.0 ) return -1.0;
  
  return -b - sqrt( h );

}

float sSphere( in vec3 ro, in vec3 rd, in vec4 sph ){
  
  vec3 oc = ro - sph.xyz;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - sph.w*sph.w;

  return step( min( -b, min( c, b*b - c ) ), 0.0 );

}

float oSphere( in vec3 pos, in vec3 nor, in vec4 sph ){

  vec3 di = sph.xyz - pos;
  float l = length(di);

  return 1.0 - max(0.0,dot(nor,di/l))*sph.w*sph.w/(l*l); 

}


float calcIntersection( in vec3 ro, in vec3 rd , in vec3 pos){
  
  const float maxd = 20.0;           // max trace distance
  const float precis = 0.001;        // precission of the intersection
  float h = precis*2.0;
  float t = 0.0;
  float res = -1.0;
  for( int i=0; i<90; i++ ){
    if( h<precis||t>maxd ) break;
    h = doModel( ro+rd*t , pos );
    t += h;
  }

  if( t<maxd ) res = t;

  return res;

}

vec3 calcNormal( in vec3 pos , in vec3 pos2 ){

  const float eps = 0.002;             // precision of the normal computation

  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1*doModel( pos + v1*eps , pos2 ) + 
                  v2*doModel( pos + v2*eps , pos2 ) + 
                  v3*doModel( pos + v3*eps , pos2 ) + 
                  v4*doModel( pos + v4*eps , pos2 ) );
}



mat3 calcLookAtMatrix( in vec3 ro, in vec3 ta, in float roll ){

  vec3 ww = normalize( ta - ro );
  vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
  vec3 vv = normalize( cross(uu,ww));

  return mat3( uu, vv, ww );

}

vec3 triplanar( in vec3 normal , in vec3 pos){

  vec3 vNorm = normal;
  vec3 vPos = pos;

  vec3 blend_weights = abs( vNorm );
  blend_weights = ( blend_weights - 0.2 ) * 7.;  
  blend_weights = max( blend_weights, 0. );
  blend_weights /= ( blend_weights.x + blend_weights.y + blend_weights.z );

  float texScale = 1.;
  float normalScale = .4;

  vec2 coord1 = vPos.yz * texScale;
  vec2 coord2 = vPos.zx * texScale;
  vec2 coord3 = vPos.xy * texScale;

  vec3 bump1 = texture2D( texture , coord1 ).rgb;  
  vec3 bump2 = texture2D( texture , coord2 ).rgb;  
  vec3 bump3 = texture2D( texture , coord3  ).rgb; 

  vec3 blended_bump = bump1 * blend_weights.xxx +  
                    bump2 * blend_weights.yyy +  
                    bump3 * blend_weights.zzz;

  vec3 tanX = vec3( vNorm.x, -vNorm.z, vNorm.y);
  vec3 tanY = vec3( vNorm.z, vNorm.y, -vNorm.x);
  vec3 tanZ = vec3(-vNorm.y, vNorm.x, vNorm.z);
  vec3 blended_tangent = tanX * blend_weights.xxx +  
                       tanY * blend_weights.yyy +  
                       tanZ * blend_weights.zzz; 

  vec3 normalTex = blended_bump * 2.0 - 1.0;
  normalTex.xy *= normalScale;
  normalTex.y *= -1.;
  normalTex = normalize( normalTex );
  mat3 tsb = mat3( normalize( blended_tangent ), normalize( cross( vNorm, blended_tangent ) ), normalize( vNorm ) );

  // vec3 bump = texture2D( tNormal , vUv ).xyz;
  vec3 finalNormal = tsb * normalTex;

  return finalNormal;


}

void main( void ){

  vec2 p = (-resolution.xy + 2.0*gl_FragCoord.xy)/resolution.y;
  vec2 m = mouse.xy/resolution.xy;

  //-----------------------------------------------------
  // camera
  //-----------------------------------------------------

  // camera movement
  vec3 ro, ta;
  doCamera( ro, ta, time, m.x );

  // camera matrix
  mat3 camMat = calcLookAtMatrix( ro, ta, 0.0 );  // 0.0 is the camera roll

  // create view ray
  vec3 rd = normalize( camMat * vec3(p.xy,2.0) ); // 2.0 is the lens length

  //-----------------------------------------------------
  // render
  //-----------------------------------------------------



  vec3 pos1 = vec3( 1. , .5 , 0. );


  float tmin = 10000.0;
  vec3  nor = vec3(0.0);
  //float occ = 1.0;
  vec3  pos = vec3(0.0);

  vec4 sph1 = vec4(-2.0, 1.0,0.0,.2);
  vec4 sph2 = vec4(0.0, 0.0,0.0,.5);
  vec4 sph3 = vec4(2.0, -1.0,0.0,.6);

  vec3 sur = vec3(1.0);

  float hit = 0.;

  vec3 sCol = vec3( 0. );
  for( int i = 0; i < 25; i++ ){

    float h = iSphere( ro , rd , spheres[i]  );

    if(  h>0.0 && h<tmin ){
      tmin = h;
      hit = 1.;
      pos = ro + h*rd;
      nor = normalize(pos-spheres[i].xyz); 
      sCol = sphereCols[i];
      //occ = oSphere( pos, nor, sph2 ) * oSphere( pos, nor, sph3 );
      sur = vec3( 1. , 1. , cos( float( i ) ) );
    }

  }


  float occ = 0.;

  vec3 occCol = vec3( 0. );

  if( hit > 0. ){

    vec3  uu  = normalize( cross( nor, vec3(0.0,1.0,1.0) ) );
    vec3  vv  = normalize( cross( uu, nor ) );
    float off = texture2D( texture , vUv * abs(sin( time )) ).x;

    //vec2 rand = hash2( time * vUv.x + vUv.y );

    off = length(hash2( time * vUv.x + vUv.y )) ;
    //off = length(hash2( vUv.x + vUv.y )) ;

    //off = 0.;
    for( int i = 0; i < 3; i++ ){

      vec2  aa = hash2( off +  float(i)*203.1 );
      float ra = sqrt(aa.y);
      float rx = ra*cos(6.2831*aa.x); 
      float ry = ra*sin(6.2831*aa.x);
      float rz = sqrt( 1.0-aa.y );
      vec3  rr = vec3( rx*uu + ry*vv + rz*nor );

      float res = 1.;

      for( int j = 0; j < 25; j++ ){

        float t = sSphere( pos , rr , spheres[j] );
        res = min( t , res );

        occCol += sphereCols[j] * ( 1. - t );

        //occCol = + 

      }

      occ += res;

    }

    occ /= 3.; //occ * occ;

    occCol /= 3.;
    // occ /= 5.;
    //occ = .0;

  }



  vec3 col =vec3(0.);// texture2D( t_audio , vec2( vUv.x , 0. ) ).xyz;//vec3(1.0);

  if( tmin<100.0 ){


    //vec3 t =  texture2D( texture , vec2( nor.x , nor.y )).xyz;
    vec3 t = triplanar( nor , pos );
    vec3 l1 = dot( t , vec3( 1., 0., 0.))* sCol;
    vec3 l2 = dot( t , vec3( -1., 0., 0.))* vec3( .4 , 1. , .6 );
  
    float l =  dot( t , vec3( -1., 0., 0.));
    vec3 a = texture2D( t_audio , vec2( l , 0. ) ).xyz;
    //col = ( l1 + l2 + a) *  occ;

    //col = sCol * occ;

    col += occCol;
    col += a;
    //col += l1 * occ;
    // col = vec3( .0 ) * occ + ( l1 + l2 + a) * ( 1. - occ );
    
  }
  //-----------------------------------------------------
  // postprocessing
  //-----------------------------------------------------
  // gamma
  //col = pow( clamp(col,0.0,1.0), vec3(0.4545) );
   
  gl_FragColor = vec4( col, 1.0 );

}
